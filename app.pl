# app.pl
use Dancer2;

my @books;
my $book_id_counter = 1;

get '/books' => sub {
    content_type 'application/json';
    return to_json(\@books);
};

sub find_book {
    my ($id) = @_;
    return (grep { $_->{id} == $id } @books)[0];
}

post '/books' => sub {
    my $data = request->body_parameters;
    return status 400, to_json({ error => 'Title and author are required.' })
        unless $data->get('title') && $data->get('author');
    my $book = {
        id     => $book_id_counter++,
        title  => $data->get('title'),
        author => $data->get('author')
    };
    push @books, $book;
    status 201;
    return $book;
};

get '/books/:id' => sub {
    my $book = find_book(route_parameters->get('id'));
    return status 404, to_json({ error => 'Book not found.' }) unless $book;
    return $book;
};

put '/books/:id' => sub {
    my $book = find_book(route_parameters->get('id'));
    return status 404, to_json({ error => 'Book not found.' }) unless $book;
    my $data = request->body_parameters;
    return status 400, to_json({ error => 'Title and author are required.' })
        unless $data->get('title') && $data->get('author');
    $book->{title}  = $data->get('title');
    $book->{author} = $data->get('author');
    return $book;
};

del '/books/:id' => sub {
    my $id = route_parameters->get('id');
    my $book = find_book($id);
    return status 404, to_json({ error => 'Book not found.' }) unless $book;
    @books = grep { $_->{id} != $id } @books;
    status 204;
    return '';
};

start;